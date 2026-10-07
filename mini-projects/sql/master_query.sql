SELECT 
    o.order_id,
    oi.order_item_id,
    p.product_id,
    o.order_status,

    pt.product_category_name_english AS product_category,
    p.product_name_lenght,
    p.product_description_lenght,
    p.product_photos_qty,

    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,

    oi.price,
    oi.freight_value,

    orr.review_score,
    orr.review_creation_date,
    orr.review_comment_title,
    orr.review_comment_message


from orders o join order_items oi on o.order_id = oi.order_id
left join products p on oi.product_id = p.product_id
left join product_category_name_translation pt on p.product_category_name = pt.product_category_name
left join(
        select
                order_id,
                review_score,
                review_creation_date,
                review_comment_title,
                review_comment_message,
                ROW_NUMBER() OVER(PARTITION BY order_id ORDER BY review_creation_date DESC) as rn
        from order_reviews orr
) as orr on o.order_id = orr.order_id and orr.rn = 1